import type { Handed } from '../../enums/Handed.js';
import { Table } from '../../enums/Table.js';
import { insertIntoDBTable } from './insertIntoDBTable.js';

interface Fields {
   baseball_reference_id: string,
   bats: Handed,
   name: string,
   throws: Handed,
   time_born: number,
}

export const insertDBPlayer = async (fields: Fields) => {
   return await insertIntoDBTable(Table.player, fields);
}