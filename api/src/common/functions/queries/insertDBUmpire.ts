import { Table } from '../../enums/Table.js';
import { insertIntoDBTable } from './insertIntoDBTable.js';

interface Fields {
   name: string,
}

export const insertDBUmpire = async (fields: Fields) => {
   return await insertIntoDBTable(Table.umpire, fields);
}