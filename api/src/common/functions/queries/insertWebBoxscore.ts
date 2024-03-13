import { Table } from '../../enums/Table.js';
import { insertIntoTable } from './insertIntoTable.js';

interface Fields {
   season: number,
   url: string,
}

export const insertWebBoxscore = async (fields: Fields) => {
   return await insertIntoTable(Table.webBoxScore, fields);
}