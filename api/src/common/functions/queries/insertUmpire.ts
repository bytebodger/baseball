import { Table } from '../../enums/Table.js';
import { insertIntoTable } from './insertIntoTable.js';

interface Fields {
   name: string,
}

export const insertUmpire = async (fields: Fields) => {
   return await insertIntoTable(Table.umpire, fields);
}